"""Sparse keyword index — the baseline every dense retriever must beat."""
from __future__ import annotations

from rank_bm25 import BM25Okapi


class BM25Index:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        # Simple whitespace tokenization is a fine starting point; you can improve
        # it later (lowercasing, stopwords) and measure whether it actually helps.
        self.tokenized = [c["text"].lower().split() for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized)

    def search(self, query: str, top_k: int) -> list[tuple[dict, float]]:
        scores = self.bm25.get_scores(query.lower().split())
        ranked = sorted(zip(self.chunks, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
