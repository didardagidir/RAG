"""Common interface so the agent can treat any retriever the same way."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    source_id: str | None = None


class Retriever(Protocol):
    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        ...
