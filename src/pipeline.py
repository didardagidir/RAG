"""Top-level wiring: build a ready-to-query agentic RAG system.

The retriever is passed in as a callable (query, k) -> list[dict]. In the notebook this
is hybrid_rerank_search; in the repo you'd wrap your index objects into the same shape.
"""
from __future__ import annotations

from src.llm import get_llm
from src.agent.graph import build_agent


class Pipeline:
    def __init__(self, agent):
        self.agent = agent

    def ask(self, question: str) -> dict:
        return self.agent.invoke({"question": question, "iterations": 0})


def build_pipeline(retriever) -> Pipeline:
    """retriever: callable (query, k) -> list of dicts with keys
    chunk_id, passage_id, title, text, score."""
    llm = get_llm()
    return Pipeline(build_agent(llm, retriever))
