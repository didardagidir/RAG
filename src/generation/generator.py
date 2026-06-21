"""Phase-2 baseline generator (non-agentic). The LLM factory now lives in src/llm.py;
re-exported here for backwards compatibility."""
from __future__ import annotations

from src.llm import get_llm, safe_invoke  # noqa: F401  (re-export)
from src.agent.prompts import GENERATE


def simple_generate(question: str, passages: list[str], llm) -> str:
    """Baseline RAG: stuff retrieved passages into the prompt and answer once.
    No agent, no loop — the number to beat before adding the agentic layer."""
    text = "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages))
    return safe_invoke(llm, GENERATE.format(question=question, passages=text))
