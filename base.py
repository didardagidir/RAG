"""Ties retrieval + generation evaluation into one report."""
from __future__ import annotations

from src.config import CONFIG
from src.evaluation import retrieval_metrics as rm


def evaluate_retrieval(records: list[dict], retriever, k: int = 10) -> dict:
    """For each record, retrieve and score against gold relevant ids.

    TODO:
      - For each record: ids = [c.chunk_id for c in retriever.retrieve(q, k)]
      - relevant = set(record['supporting_ids'])
      - compute recall/mrr/ndcg per query, then rm.aggregate(...)
    """
    raise NotImplementedError
