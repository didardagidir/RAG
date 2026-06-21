"""Agentic RAG — interactive demo (HotpotQA multi-hop).

What makes this demo worth building: it surfaces the agent's *reasoning trace* —
sub-questions, retrieval, the grade decision, and any rewrite loops — not just a final
answer. That transparency is the whole point of an agentic system: it's auditable, not
a black box.

Run locally:
    streamlit run app/streamlit_app.py

Requires (placed in DATA_DIR, default ./data — these are produced by the notebook and
saved to disk):
    hotpot_chunks.json, hotpot_embeddings.npy, hotpot_queries.json, hotpot_qrels.json
Plus a GEMINI_API_KEY in your .env (free tier: aistudio.google.com).
"""
from __future__ import annotations

import os
import json
import numpy as np
import streamlit as st

# repo imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm import get_llm
from src.retrieval.searchers import build_searchers
from src.agent.graph import build_agent

DATA_DIR = os.environ.get("DATA_DIR", "./data")

st.set_page_config(page_title="Agentic RAG — Multi-Hop QA", layout="wide")


# ---------- expensive setup, cached so it runs once ----------
@st.cache_resource(show_spinner="Loading models and index (first run only)...")
def load_everything():
    import torch
    from sentence_transformers import SentenceTransformer, CrossEncoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device, max_length=512)

    chunks = json.load(open(f"{DATA_DIR}/hotpot_chunks.json"))
    embeddings = np.load(f"{DATA_DIR}/hotpot_embeddings.npy")
    queries = json.load(open(f"{DATA_DIR}/hotpot_queries.json"))

    searchers = build_searchers(chunks, embeddings, model, reranker)
    llm = get_llm()
    agent = build_agent(llm, searchers["hybrid_rerank"])
    return agent, queries


def run_with_trace(agent, question):
    """Stream the agent and collect each node's update for display."""
    trace, final = [], {}
    for step in agent.stream({"question": question, "iterations": 0}):
        for node, update in step.items():
            trace.append((node, update))
            final.update(update)
    return trace, final


# ---------- UI ----------
st.title("Agentic RAG — Multi-Hop Question Answering")
st.caption("LangGraph agent over a HotpotQA corpus. Watch it decompose, retrieve, "
           "grade its own evidence, and synthesize a grounded answer.")

try:
    agent, queries = load_everything()
except FileNotFoundError as e:
    st.error(f"Missing data file: {e}. Place the hotpot_* files in {DATA_DIR}/ "
             "(produced by the notebook).")
    st.stop()

# example questions from the dataset (multi-hop)
examples = list(queries.values())[:6]
with st.expander("Try an example multi-hop question"):
    for ex in examples:
        if st.button(ex, key=ex):
            st.session_state["q"] = ex

question = st.text_input("Ask a question", value=st.session_state.get("q", ""))

if st.button("Run agent", type="primary") and question:
    with st.spinner("Agent is reasoning..."):
        trace, final = run_with_trace(agent, question)

    # final answer up top
    st.subheader("Answer")
    st.write(final.get("answer", "(no answer produced)"))

    # sources
    sources = final.get("sources", [])
    if sources:
        st.subheader("Sources")
        for s in sources:
            st.markdown(f"- **[{s['n']}]** {s['title']}  ·  `{s['passage_id']}`")

    # the reasoning trace — the part that makes this 'agentic' visible
    st.subheader("Reasoning trace")
    st.caption(f"Total retrieve→grade iterations: {final.get('iterations', '?')}")
    for node, upd in trace:
        if node == "decompose":
            st.markdown(f"**Decompose** → {len(upd['sub_questions'])} sub-question(s)")
            for sq in upd["sub_questions"]:
                st.markdown(f"&nbsp;&nbsp;• {sq}", unsafe_allow_html=True)
        elif node == "retrieve":
            st.markdown(f"**Retrieve** → {len(upd['retrieved'])} passages accumulated")
        elif node == "grade":
            verdict = "✅ sufficient" if upd.get("sufficient") else "❌ insufficient"
            st.markdown(f"**Grade** → {verdict}")
            if upd.get("missing"):
                st.markdown(f"&nbsp;&nbsp;_missing: {upd['missing']}_", unsafe_allow_html=True)
        elif node == "rewrite":
            st.markdown(f"**Rewrite** → new query: _{upd['sub_questions'][0]}_")
        elif node == "generate":
            st.markdown("**Generate** → grounded answer produced")
