"""Assembles the nodes into a LangGraph state machine.

The conditional edge `_should_continue` is what turns a linear pipeline into an agent:
after grading, it either loops back (rewrite -> retrieve) to gather more evidence, or
moves on to generate. MAX_ITER (from config) is the safety valve against infinite loops.
"""
from __future__ import annotations

from functools import partial
from langgraph.graph import StateGraph, END

from src.config import CONFIG
from src.agent.state import AgentState
from src.agent import nodes


def _should_continue(state: AgentState) -> str:
    if state.get("sufficient"):
        return "generate"
    if state.get("iterations", 0) >= CONFIG.agent["max_iterations"]:
        return "generate"          # give up gracefully, answer with what we have
    return "rewrite"               # insufficient + budget left -> try again


def build_agent(llm, retriever):
    """retriever: callable (query, k) -> list[dict]. Returns a compiled agent."""
    g = StateGraph(AgentState)
    g.add_node("decompose", partial(nodes.decompose, llm=llm))
    g.add_node("retrieve", partial(nodes.retrieve, retriever=retriever))
    g.add_node("grade", partial(nodes.grade, llm=llm))
    g.add_node("rewrite", partial(nodes.rewrite, llm=llm))
    g.add_node("generate", partial(nodes.generate, llm=llm))

    g.set_entry_point("decompose")
    g.add_edge("decompose", "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", _should_continue,
                            {"rewrite": "rewrite", "generate": "generate"})
    g.add_edge("rewrite", "retrieve")     # the loop
    g.add_edge("generate", END)
    return g.compile()
