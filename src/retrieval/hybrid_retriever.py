"""Combine BM25 (sparse) and dense results with Reciprocal Rank Fusion.

RRF is a simple, strong way to merge two ranked lists without tuning weights:
    score(d) = sum over lists of 1 / (rrf_k + rank_of_d_in_list)

This is one of the highest-value/lowest-effort pieces — implement it and measure
the lift over either retriever alone. That delta is a great thing to put in your README.
"""
from __future__ import annotations

from src.config import CONFIG
from src.retrieval.base import RetrievedChunk


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedChunk]], rrf_k: int | None = None
) -> list[RetrievedChunk]:
    rrf_k = rrf_k or CONFIG.retrieval["rrf_k"]
    scores: dict[str, float] = {}
    lookup: dict[str, RetrievedChunk] = {}
    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            lookup[chunk.chunk_id] = chunk
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    out = []
    for cid, s in fused:
        c = lookup[cid]
        out.append(RetrievedChunk(c.chunk_id, c.text, s, c.source_id))
    return out


class HybridRetriever:
    """Wires BM25 + dense together. TODO: inject your index objects and
    convert their raw results into RetrievedChunk before fusing."""

    def __init__(self, bm25_index, vector_index):
        self.bm25 = bm25_index
        self.vector = vector_index

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        # TODO: call both indexes, wrap results as RetrievedChunk lists, RRF-fuse,
        # return top-k.
        raise NotImplementedError
