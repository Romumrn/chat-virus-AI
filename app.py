#!/usr/bin/python3
import streamlit as st
import pandas as pd
import requests
import os
from tools import wikipedia_search, query_dataframe, create_visualization, TOOLS_SPEC

# ==================== CONSTANTS ==================== #
TAXO_DB_PATH = "data/viral_taxo.csv"
HOST_DB_PATH = "data/virushostdb.tsv"
OLLAMA_BASE_URL = "http://localhost:11434"
PAGE_TITLE = "Virus Dataset AI Agent 🦠"

DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_REPEAT_PENALTY = 1.0
DEFAULT_SEED = 42
DEFAULT_MAX_TOOL_CALLS = 7
DEFAULT_PREVIEW_ROWS = 50
DEFAULT_WIKIPEDIA_LIMIT = 5000


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


# ==================== AGENT LOOP ==================== #

def ollama_agent_loop(model, user_query, df_taxo, df_host, temperature=DEFAULT_TEMPERATURE,
                      top_p=DEFAULT_TOP_P, repeat_penalty=DEFAULT_REPEAT_PENALTY, seed=DEFAULT_SEED,
                      max_tool_calls=DEFAULT_MAX_TOOL_CALLS, preview_rows=DEFAULT_PREVIEW_ROWS,
                      wikipedia_limit=DEFAULT_WIKIPEDIA_LIMIT):
    df_taxo_columns_str = ", ".join(df_taxo.columns)
    df_host_columns_str = ", ".join(df_host.columns)

    messages = [
        {
            "role": "system",
            "content": f"""
            You are a scientific bioinformatics assistant specialized in viruses.

You work with:

- A pandas DataFrame called `df_taxo` containing viral taxonomy data.
  Available columns:
  { df_taxo_columns_str}

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

    used_sources, used_wikipedia_urls, executed_codes, generated_figures = set(), [], [], []
    tool_call_count = 0

    print( "========= USER Question =========")
    print( f"**Active config** — temp `{temperature}` · top_p `{top_p}` ·  penalty `{repeat_penalty}` · seed `{seed}` ·  max calls `{max_tool_calls}` · preview `{preview_rows}` rows")
    print( "Question : ", user_query )
    while True:
        r = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": model, "messages": messages, "tools": TOOLS_SPEC, "stream": False,
                "options": {"temperature": temperature, "top_p": top_p, "repeat_penalty": repeat_penalty, "seed": seed}
            },
            timeout=120
        )
        r.raise_for_status()
        msg = r.json()["message"]

        try:
            print("thinking : ", msg['thinking'], "\n" )
        except:
            pass
        
        # CONDITION d4Arret 
        if "tool_calls" not in msg:
            messages.append(msg)
            print( "RESULT :", msg["content"] )
            return msg["content"], generated_figures, used_sources, used_wikipedia_urls, executed_codes

        messages.append({"role": "assistant", "content": "", "tool_calls": msg["tool_calls"]})

        if tool_call_count >= max_tool_calls:
            print( "MAX CALL")
            messages.append({
                "role": "system",
                "content": f"Tool call limit reached ({max_tool_calls}). Synthesize a final answer to: '{user_query}'"
            })
            
        for call in msg["tool_calls"]:
            print("Tool call : ", call, "\n" )
            tool_call_count += 1
            name = call["function"]["name"]
            args = call["function"]["arguments"]

            if name == "wikipedia_search":
                output = wikipedia_search(**args, wikipedia_limit=wikipedia_limit)
                if output["success"]:
                    used_sources.add("Wikipedia")
                    used_wikipedia_urls.append(output["url"])
                    content = f"**{output['title']}**\n\n{output['extract']}\n\n🔗 {output['url']}"
                else:
                    content = output["message"]

            elif name == "query_dataframe":
                output = query_dataframe(args["code"], df_taxo, df_host, preview_rows=preview_rows)
                if output["success"]:
                    executed_codes.append(args["code"])
                    used_sources.add("Dataset query")
                    content = f"Query OK. Shape: {output['shape']}\nColumns: {', '.join(output['columns'])}\n{output['preview']}"
                else:
                    content = f"Error:\n{output['message']}"

            elif name == "create_visualization":
                output = create_visualization(args["code"], df_taxo, df_host)
                if output["success"]:
                    executed_codes.append(args["code"])
                    used_sources.add("Dataset visualization")
                    generated_figures.append(output["figure"])
                    content = "Visualization created successfully."
                else:
                    content = f"Error:\n{output['message']}"

            else:
                content = f"Unknown tool: {name}"

            messages.append({"role": "tool", "tool_call_id": call["id"], "name": name, "content": content})




# ==================== MAIN ==================== #

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
        For best results:
        - Use English whenever possible
        - Ask precise and clearly formulated questions
        - Provide as much relevant detail as you can
        - Double-check that your question is scientifically accurate

        Example questions:   
        - “Give me information about Orthopoxvirus. Is it a family or a genus? How many species does it include?”
        - “I want more information about polyomavirus”
        - “Show a pie chart of genus distribution within Poxviridae.”
        - “What are the known hosts of Orthopoxvirus Abatino?”     
         
    """)

    # ── Sidebar ──
    with st.sidebar:
        st.title("Virus Dataset AI Agent 🦠 ")
        st.caption(
            """
            Explore Viral Taxonomy with AI-Assisted Analysis
This agent helps you explore the viral world through structured datasets and AI-guided analysis. You can ask about:
- Virus families, genera, and species
- Host information
- Taxonomic relationships
- Data visualizations
- And more ...

You have access to curated viral databases, including:
- Viral taxonomy data from SRA/NCBI
- Viral host information from genome.jp
- Wikipedia

Additional databases will be integrated soon and it will be amazing !
"""
        )
         
        st.header("Settings")

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

        model_names = [m["name"] for m in requests.get(f"{OLLAMA_BASE_URL}/api/tags").json().get("models", [])]
        if not model_names:
            st.error("No models available in Ollama")
            st.stop()

        with st.expander("⚙️ Expert mode", expanded=False):
            default_index = next((i for i, m in enumerate(model_names) if m.startswith("gpt-oss")), 0)
            model = st.selectbox("LLM Model", options=model_names, index=default_index,
                                help="Select a model able to use tools")

        
            st.markdown("**Sampling**")
            temperature    = st.slider("Temperature",    0.0, 2.0,   DEFAULT_TEMPERATURE,    0.05)
            top_p          = st.slider("Top-p",          0.0, 1.0,   DEFAULT_TOP_P,          0.05)
            repeat_penalty = st.slider("Repeat penalty", 0.5, 2.0,   DEFAULT_REPEAT_PENALTY, 0.05)
            seed           = st.number_input("Seed", min_value=-1, max_value=99999, value=DEFAULT_SEED, step=1)
            st.markdown("**Agent**")
            max_tool_calls  = st.slider("Max tool calls",  1,   20,    DEFAULT_MAX_TOOL_CALLS)
            preview_rows    = st.slider("Preview rows",    5,   200,   DEFAULT_PREVIEW_ROWS,   5)
            wikipedia_limit = st.slider("Wiki limit",      500, 20000, DEFAULT_WIKIPEDIA_LIMIT, 500)

        st.markdown("---")
        # st.markdown(
        #     f"**Active config** — temp `{temperature}` · top_p `{top_p}` · "
        #     f"penalty `{repeat_penalty}` · seed `{seed}` · "
        #     f"max calls `{max_tool_calls}` · preview `{preview_rows}` rows"
        # )
        st.caption("🔗 GitHub: https://github.com/Romumrn/chat-virus-AI")


    # ── Session state ──
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ── Chat display ──
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            for fig in msg.get("figures", []):
                st.plotly_chart(fig, width='stretch')
            if msg.get("wikipedia_urls") or msg.get("executed_codes"):
                with st.expander("📚 Sources"):
                    if msg.get("wikipedia_urls"):
                        st.markdown("**📘 Wikipedia**")
                        for url in msg["wikipedia_urls"]:
                            title = url.split('/')[-1].replace('_', ' ')
                            st.markdown(f"- [{title}]({url})")
                    if msg.get("executed_codes"):
                        st.markdown("**📊 Dataset Query & Visualization**")
                        full_code = "\n\n---\n\n".join(
                            f"# Code {i}\n{code}"
                            for i, code in enumerate(msg["executed_codes"], 1)
                        )
                        st.code(full_code, language="python")

    # ── Chat input ──
    if query := st.chat_input("Ask about viruses..."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, figures, used_sources, wikipedia_urls, executed_codes = ollama_agent_loop(
                    model=model,
                    user_query=query,
                    df_taxo=df_taxo,
                    df_host=df_host,
                    temperature=temperature,
                    top_p=top_p,
                    repeat_penalty=repeat_penalty,
                    seed=seed,
                    max_tool_calls=max_tool_calls,
                    preview_rows=preview_rows,
                    wikipedia_limit=wikipedia_limit,
                )
            st.markdown(answer)
            for fig in figures:
                st.plotly_chart(fig, width='stretch')
                
            if wikipedia_urls or executed_codes:
                with st.expander("📚 Sources"):
                    if wikipedia_urls:
                        st.markdown("**📘 Wikipedia**")
                        for url in wikipedia_urls:
                            title = url.split('/')[-1].replace('_', ' ')
                            st.markdown(f"- [{title}]({url})")
                    
                    if executed_codes:
                        st.markdown("**📊 Dataset Query & Visualization**")
                        full_code = "\n\n---\n\n".join(
                            f"# Code {i}\n{code}"
                            for i, code in enumerate(executed_codes, 1)
                        )
                        st.code(full_code, language="python")


        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "figures": figures,
            "wikipedia_urls": wikipedia_urls,
            "executed_codes": executed_codes,
        })

    st.markdown("---")
    st.caption("⚠️ **AI is not magic** — Results may contain errors and should be verified for scientific or medical use.")


if __name__ == "__main__":
    main()