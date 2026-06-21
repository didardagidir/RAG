"""Cross-encoder reranker: reorders the top-k candidates by joint query-passage
relevance. More accurate than embedding similarity, but slower — so you only run
it on the ~20 candidates retrieval already shortlisted."""
from __future__ import annotations

from src.config import CONFIG
from src.retrieval.base import RetrievedChunk


class Reranker:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or CONFIG.retrieval["reranker_model"]
        # TODO: load sentence_transformers.CrossEncoder(self.model_name)

    def rerank(self, query: str, chunks: list[RetrievedChunk], final_k: int) -> list[RetrievedChunk]:
        # TODO: score [query, chunk.text] pairs with the cross-encoder, sort desc,
        # return top final_k with updated scores.
        raise NotImplementedError
