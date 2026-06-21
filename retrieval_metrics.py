"""The reasoning nodes. Each takes the agent state, does its work via the LLM and/or
retriever, and returns only the state fields it changed.

All LLM calls go through safe_invoke() for rate-limit resilience. decompose() skips the
LLM entirely on obviously-simple questions (a zero-cost heuristic) to conserve quota.
"""
from __future__ import annotations

import json
import re

from src.config import CONFIG
from src.llm import safe_invoke
from src.agent import prompts
from src.agent.state import AgentState

# multi-hop markers: if NONE are present we treat the question as simple and skip
# the decompose LLM call. Conservative by design — a stray marker just costs one call,
# whereas wrongly calling a multi-hop question "simple" would lose decomposition.
_MULTIHOP = [r"\bthat\b", r"\bwhich\b", r"\bwhose\b", r"who .*\bof\b",
             r"founded .*(company|by)", r"acquired"]


def _looks_simple(q: str) -> bool:
    return not any(re.search(p, q.lower()) for p in _MULTIHOP)


def _parse_json(raw: str, fallback):
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return fallback


def decompose(state: AgentState, llm) -> AgentState:
    q = state["question"]
    if _looks_simple(q):
        return {"sub_questions": [q]}                      # no LLM call
    raw = safe_invoke(llm, prompts.DECOMPOSE.format(question=q))
    subs = _parse_json(raw, [q])
    subs = subs if isinstance(subs, list) and subs else [q]
    return {"sub_questions": subs[: CONFIG.agent["max_subquestions"]]}


def retrieve(state: AgentState, retriever) -> AgentState:
    """retriever is a callable: (query, k) -> list[dict] with chunk_id/passage_id/etc."""
    existing = {r["chunk_id"]: r for r in state.get("retrieved", [])}
    k = CONFIG.retrieval["final_k"]
    for sq in state["sub_questions"]:
        for r in retriever(sq, k):
            existing[r["chunk_id"]] = r
    return {"retrieved": list(existing.values()),
            "iterations": state.get("iterations", 0) + 1}


def grade(state: AgentState, llm) -> AgentState:
    passages = "\n\n".join(
        f"[{i+1}] {r.get('title','')}: {r['text'][:400]}"
        for i, r in enumerate(state["retrieved"])
    )
    raw = safe_invoke(llm, prompts.GRADE.format(question=state["question"], passages=passages))
    result = _parse_json(raw, {"sufficient": True, "missing": None})  # default: stop the loop
    return {"sufficient": bool(result.get("sufficient", True)),
            "missing": result.get("missing")}


def rewrite(state: AgentState, llm) -> AgentState:
    new_q = safe_invoke(llm, prompts.REWRITE.format(
        question=state["question"], missing=state.get("missing")))
    return {"sub_questions": [new_q]}


def generate(state: AgentState, llm) -> AgentState:
    passages = "\n\n".join(
        f"[{i+1}] {r.get('title','')}: {r['text']}"
        for i, r in enumerate(state["retrieved"])
    )
    answer = safe_invoke(llm, prompts.GENERATE.format(question=state["question"], passages=passages))
    sources = [{"n": i+1, "title": r.get("title", ""), "passage_id": r.get("passage_id")}
               for i, r in enumerate(state["retrieved"])]
    return {"answer": answer, "sources": sources}
