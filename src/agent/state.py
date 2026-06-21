"""State that flows through the agent graph. Each node reads what it needs and
returns only the fields it changed; LangGraph merges the updates."""
from __future__ import annotations

from typing import TypedDict, List, Optional


class AgentState(TypedDict, total=False):
    question: str                 # original user question
    sub_questions: List[str]      # decompose output (or rewrite's new query)
    retrieved: List[dict]         # accumulated evidence chunks
    sufficient: bool              # grade decision
    missing: Optional[str]        # grade's explanation of what's missing
    iterations: int               # number of retrieve->grade loops
    answer: str                   # final grounded answer
    sources: List[dict]           # passage sources backing the answer
