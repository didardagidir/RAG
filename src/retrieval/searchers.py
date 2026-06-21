"""Retriever factory: builds BM25 / dense / hybrid / hybrid+rerank search functions
over a given (chunks, embeddings) pair.

This is the proven implementation used throughout the project. Keeping it as a factory
means the same code serves multiple corpora (NQ, HotpotQA) — pass different chunks +
embeddings and get an independent set of searchers.

Each searcher returns a list of dicts: {chunk_id, passage_id, title, text, score}.
The agent treats `hybrid_rerank` as its retriever (a callable (query, k) -> list[dict]).
"""
from __future__ import annotations

import re
import numpy as np
from rank_bm25 import BM25Okapi


def _tok(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def build_searchers(chunks: list[dict], embeddings, model, reranker, rrf_k: int = 60):
    """model: SentenceTransformer (for query embedding).
    reranker: CrossEncoder (for reranking). Returns a dict of callables."""
    emb = np.asarray(embeddings, dtype=np.float32)
    ids = [c["chunk_id"] for c in chunks]
    pid = [c["passage_id"] for c in chunks]
    title = [c.get("title", "") for c in chunks]
    text = [c["text"] for c in chunks]
    bm25 = BM25Okapi([_tok(c["text"]) for c in chunks])

    def _row(i, score):
        return {"chunk_id": ids[i], "passage_id": pid[i], "title": title[i],
                "text": text[i], "score": float(score)}

    def dense(q, k=5):
        qv = model.encode([q], normalize_embeddings=True)[0]
        sims = emb @ qv
        return [_row(i, sims[i]) for i in np.argsort(-sims)[:k]]

    def bm25_search(q, k=5):
        sc = bm25.get_scores(_tok(q))
        return [_row(i, sc[i]) for i in np.argsort(-sc)[:k]]

    def _rrf(lists):
        scores, lookup = {}, {}
        for L in lists:
            for rank, x in enumerate(L):
                scores[x["chunk_id"]] = scores.get(x["chunk_id"], 0.0) + 1.0 / (rrf_k + rank)
                lookup[x["chunk_id"]] = x
        return [{**lookup[i], "score": scores[i]}
                for i in sorted(scores, key=scores.get, reverse=True)]

    def hybrid(q, k=5, pool=30):
        return _rrf([dense(q, pool), bm25_search(q, pool)])[:k]

    def hybrid_rerank(q, k=5, pool=30):
        cands = hybrid(q, pool)
        pairs = [[q, c["text"]] for c in cands]
        sc = reranker.predict(pairs)
        order = np.argsort(-sc)
        out = []
        for idx in order[:k]:
            c = dict(cands[idx])
            c["score"] = float(sc[idx])
            out.append(c)
        return out

    return {"dense": dense, "bm25": bm25_search,
            "hybrid": hybrid, "hybrid_rerank": hybrid_rerank}
